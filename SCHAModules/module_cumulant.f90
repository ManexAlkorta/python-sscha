module module_cumulant

use omp_lib
implicit none

contains

subroutine get_bubble(norbitq3,I3,osq,deltaF, &
nrefq3,n_mode)
    integer, dimension(nrefq3), intent(in) :: norbitq3
    double precision, dimension(nrefq3,n_mode,n_mode,n_mode), intent(in) :: I3
    complex, dimension(nrefq3,n_mode,n_mode,n_mode), intent(in) :: osq

    double precision, intent(out) :: deltaF
    
    double precision, dimension(n_mode,n_mode,n_mode) :: osq2
    integer :: nrefq3, n_mode, rq3

    deltaF = 0.0d0
    do rq3 = 1, nrefq3
        osq2 = real(conjg(osq(rq3,:,:,:))*osq(rq3,:,:,:))
        deltaF = deltaF + norbitq3(rq3)*sum(I3(rq3,:,:,:)*osq2)
    end do
    deltaF = -1.0d0/48.0d0*deltaF
end subroutine

subroutine get_V(d3,lq,wq,V,&
n_mode_sc,nq,n_mode)
    double precision, dimension(n_mode_sc,n_mode_sc,n_mode_sc), intent(in) :: d3
    complex, dimension(nq, n_mode, n_mode_sc), intent(in) :: lq
    double precision, dimension(nq, n_mode), intent(in) :: wq

    complex, dimension(nq,nq,nq,n_mode,n_mode,n_mode), intent(out) :: V

    integer :: n_mode,n_mode_sc,nq
    integer :: q1,q2,q3,mu1,mu2,mu3,a,b,c
    double precision :: ktea

    V = (0.0d0,0.0d0)

    do q1 = 1, nq
        print*, q1
        do q2 = 1, nq
            !$omp parallel private (mu1,mu2,mu3)
            !$omp do schedule (static) private (a,b,c)
            do q3 = 1, nq
                do mu1 = 1, n_mode
                    do mu2 = 1, n_mode
                        do mu3 = 1, n_mode
                            ktea = 1/(sqrt(8.0d0*wq(q1,mu1)*wq(q2,mu2)*wq(q3,mu3)))
                            do a = 1, n_mode_sc
                                do b = 1, n_mode_sc
                                    do c = 1, n_mode_sc
                                        V(q1,q2,q3,mu1,mu2,mu3) = &
V(q1,q2,q3,mu1,mu2,mu3) + &
ktea*d3(a,b,c)*lq(q1,mu1,a)*lq(q2,mu2,b)*lq(q3,mu3,c)
                                    end do
                                end do
                            end do
                        end do
                    end do
                end do
            end do
            !$omp end do
            !$omp end parallel
        end do
    end do
end subroutine
end module